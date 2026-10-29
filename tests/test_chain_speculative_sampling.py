# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness test for sampling_grammar/chain_speculative_sampling."""

import importlib
import json

import flaggems_sglang
import pytest
import torch
import triton
import triton.language as tl
from flaggems_sglang.ops.chain_speculative_sampling import base_group_prefix
from flaggems_sglang.reference import get_reference

# ---------------------------------------------------------------------------
# Tolerance helper (from kernel-comp-baseline/harness/correctness.py)
# ---------------------------------------------------------------------------

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype, _DEFAULT_TOLERANCE
        )
    )
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=True,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/sampling_grammar/chain_speculative_sampling/cases.py)
# ---------------------------------------------------------------------------


def _case(B, S, V, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)

    candidates = torch.randint(
        0,
        V,
        (B, S),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    retrive_index = torch.arange(
        B * S, dtype=torch.int64, device=flaggems_sglang.device
    ).reshape(B, S)
    uniform_samples = torch.rand(
        B, S - 1, generator=g, device=flaggems_sglang.device
    )
    uniform_samples_for_final_sampling = torch.rand(
        B, generator=g, device=flaggems_sglang.device
    )

    target_logits = torch.randn(
        B, S, V, generator=g, device=flaggems_sglang.device
    )
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_logits = torch.randn(
        B, S - 1, V, generator=g, device=flaggems_sglang.device
    )
    draft_probs = torch.softmax(draft_logits, dim=-1)

    return dict(
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        # Precomputed here (not derived via .item() inside baseline/solution)
        # since retrive_index = arange(B*S) makes it exact, and a host sync
        # inside the timed function breaks CUDA-graph capture in bench.py.
        num_slots=B * S,
        check=_check,
    )


def _check(actual, expected):
    a_pred, a_idx, a_num = actual
    e_pred, e_idx, e_num = expected
    assert torch.equal(a_pred, e_pred), "predicts mismatch"
    assert torch.equal(a_idx, e_idx), "accept_index mismatch"
    assert torch.equal(a_num, e_num), "accept_token_num mismatch"


_CASE_FACTORIES = [
    lambda: _case(4, 5, 32, seed=0),
    lambda: _case(4, 5, 32, seed=1),
    lambda: _case(8, 8, 128, seed=2),
    lambda: _case(2, 3, 16, seed=3),
]
_CASE_FACTORIES += [
    (lambda b=b: _case(b, 4, 32000, seed=0)) for b in (1, 8, 64, 512, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.chain_speculative_sampling
def test_chain_speculative_sampling(case_idx):
    # CI control: execute the unchanged native test body from master.
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("chain_speculative_sampling")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    # Reference
    expected = reference(**kwargs)

    # Operator under test
    try:
        from flaggems_sglang import chain_speculative_sampling
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "sampling_grammar/chain_speculative_sampling ops module not found"
        )
        return

    try:
        actual = chain_speculative_sampling(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "sampling_grammar/chain_speculative_sampling not yet implemented"
        )
        return

    # Temporary CI diagnostics only; original assertions remain below.
    if case_idx == 8:
        try:
            _chain_diagnose(kwargs, actual, expected)
        except Exception as diagnostic_error:
            import traceback

            print(
                "[CHAIN-DIAG-20261029] diagnostic error:",
                repr(diagnostic_error),
                flush=True,
            )
            traceback.print_exc()

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if a is None and e is None:
                continue
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check


def _chain_emit(label, **values):
    print(
        "[CHAIN-DIAG-20261029] " + json.dumps(dict(label=label, **values)),
        flush=True,
    )


@triton.jit
def _chain_trace_cdf(
    Values, Totals, Trace, V: tl.constexpr, ITEMS: tl.constexpr
):
    # Mirror base_sample_kernel's prefix operations without changing the real call.
    tile = tl.program_id(0)
    t = tl.arange(0, 128)
    dtype: tl.constexpr = Values.dtype.element_ty
    prefix, unused = base_group_prefix(Values, 0, tile, V, ITEMS)
    carry = tl.full((), 0.0, tl.float32)
    for window in range(tl.cdiv(tile, 32)):
        previous = tile - 1 - window * 32 - t
        partial = tl.load(Totals + previous, (previous >= 0) & (t < 32), 0).to(
            tl.float32
        )
        for j in tl.static_range(5):
            other = tl.gather(partial, tl.minimum(t + (1 << j), 127), 0)
            partial = (partial + other).to(dtype).to(tl.float32)
        window_sum = tl.sum(tl.where(t == 0, partial, 0.0), 0)
        carry = (window_sum + carry).to(dtype).to(tl.float32)
    current = (carry + prefix).to(dtype).to(tl.float32)
    for k in tl.static_range(ITEMS):
        idx = tile * 128 * ITEMS + t * ITEMS + k
        val = tl.load(Values + idx, idx < V, 0).to(tl.float32)
        current = (current + val).to(dtype).to(tl.float32)
        tl.store(Trace + idx, current, idx < V)


def _chain_token(cdf, threshold):
    matches = torch.nonzero(cdf > threshold).flatten()
    return int(matches[0].item()) if matches.numel() else cdf.numel() - 1


def _chain_diagnose(kwargs, actual, expected):
    from flaggems_sglang.reference_candidate import get_candidate

    native = importlib.import_module(
        "flaggems_sglang.ops.chain_speculative_sampling"
    )
    public = flaggems_sglang.chain_speculative_sampling
    helper = get_candidate(
        "chain_speculative_sampling",
        reference=get_reference("chain_speculative_sampling"),
    )
    torch.cuda.synchronize()
    differences = [
        (a != e).nonzero().cpu().tolist() for a, e in zip(actual, expected)
    ]
    _chain_emit(
        "outputs",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=triton.__version__,
        gpu=torch.cuda.get_device_name(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        candidate_module=public.__module__,
        helper_is_public=helper is public,
        output_mismatch_counts=[len(d) for d in differences],
        first_mismatches=[d[:8] for d in differences],
    )
    # Same tensors and original public function; check repeatability and helper identity.
    for repeat in range(2):
        repeated = helper(**kwargs)
        torch.cuda.synchronize()
        _chain_emit(
            "repeat",
            repeat=repeat,
            versus_original=[
                int((a != b).sum().item()) for a, b in zip(repeated, actual)
            ],
            versus_reference=[
                int((a != e).sum().item()) for a, e in zip(repeated, expected)
            ],
        )
        del repeated

    if not differences[0]:
        return
    batch, steps = kwargs["candidates"].shape
    vocab = kwargs["target_probs"].shape[-1]
    rows = sorted({pos[0] // steps for pos in differences[0]})
    for b in rows[:4]:
        rn = int(expected[2][b].item())
        an = int(actual[2][b].item())
        reference_slot = int(expected[1][b, rn].item())
        actual_slot = int(actual[1][b, an].item())
        _chain_emit(
            "row",
            b=b,
            reference_accepted=rn,
            actual_accepted=an,
            reference_index=expected[1][b].cpu().tolist(),
            actual_index=actual[1][b].cpu().tolist(),
            reference_final_slot=reference_slot,
            actual_final_slot=actual_slot,
            reference_token=int(expected[0][reference_slot].item()),
            actual_token=int(actual[0][actual_slot].item()),
        )
        if rn != an or not torch.equal(actual[1][b], expected[1][b]):
            _chain_emit(
                "acceptance_diff",
                b=b,
                candidate_tokens=kwargs["candidates"][b].cpu().tolist(),
                coins=kwargs["uniform_samples"][b].cpu().tolist(),
            )
            for step in range(1, steps):
                token = int(kwargs["candidates"][b, step].item())
                p = kwargs["target_probs"][b, step - 1, token]
                q = kwargs["draft_probs"][b, step - 1, token]
                u = kwargs["uniform_samples"][b, step - 1]
                _chain_emit(
                    "acceptance_threshold",
                    b=b,
                    step=step,
                    token=token,
                    p=float(p.item()),
                    q=float(q.item()),
                    coin=float(u.item()),
                    product=float((u * q).item()),
                    margin=float((p - u * q).item()),
                )
            continue

        p = kwargs["target_probs"][b, rn]
        val = (
            p.clone()
            if rn == steps - 1
            else (
                p - torch.nan_to_num(kwargs["draft_probs"][b, rn], nan=0.0)
            ).clamp(min=0.0)
        )
        device = p.device
        values = torch.empty((1, vocab), device=device, dtype=p.dtype)
        native_sum = torch.empty((1,), device=device, dtype=p.dtype)
        predicts = torch.empty((1,), device=device, dtype=torch.int32)
        slots = torch.zeros((1,), device=device, dtype=torch.int32)
        row = torch.tensor([rn], device=device, dtype=torch.int32)
        accepted = torch.tensor(
            [int(rn == steps - 1)], device=device, dtype=torch.int32
        )
        native.base_values_kernel[(1, triton.cdiv(vocab, 256))](
            values,
            kwargs["target_probs"][b : b + 1],
            kwargs["draft_probs"][b : b + 1],
            row,
            accepted,
            S=steps,
            V=vocab,
            FMAX=torch.finfo(p.dtype).max,
            BLOCK=256,
            enable_fp_fusion=False,
        )
        nt = min(
            512, 1 << ((vocab // 4 if vocab > 128 else vocab).bit_length() - 1)
        )
        native.base_sum_kernel[(1,)](
            values,
            native_sum,
            predicts,
            slots,
            vocab,
            nt,
            nt.bit_length() - 1,
            min(5, nt.bit_length() - 1),
            enable_fp_fusion=False,
        )
        items = 15 if p.dtype == torch.float32 else 30
        tiles = triton.cdiv(vocab, 128 * items)
        totals = torch.empty((1, tiles), device=device, dtype=p.dtype)
        trace = torch.empty((vocab,), device=device, dtype=p.dtype)
        native.base_tile_totals_kernel[(1, tiles)](
            values, totals, vocab, items, enable_fp_fusion=False
        )
        _chain_trace_cdf[(tiles,)](
            values, totals, trace, vocab, items, enable_fp_fusion=False
        )
        coin = kwargs["uniform_samples_for_final_sampling"][b]
        native.base_sample_kernel[(1, tiles)](
            values,
            totals,
            native_sum,
            coin.reshape(1),
            predicts,
            slots,
            vocab,
            items,
            False,
            enable_fp_fusion=False,
        )
        torch.cuda.synchronize()
        ref_sum = val.sum()
        ref_cdf = val.cumsum(0)
        rt = coin * ref_sum
        ntarget = coin * native_sum[0]
        ref_token = int(expected[0][reference_slot].item())
        actual_token = int(actual[0][actual_slot].item())
        traced_token = _chain_token(trace, ntarget)
        isolated_token = int(predicts[0].item())
        mixed = {
            "torch_cdf_torch_sum": _chain_token(ref_cdf, rt),
            "torch_cdf_native_sum": _chain_token(ref_cdf, ntarget),
            "native_cdf_torch_sum": _chain_token(trace, rt),
            "native_cdf_native_sum": traced_token,
            "float64_analysis": _chain_token(
                val.double().cumsum(0), coin.double() * val.double().sum()
            ),
        }
        around = sorted(
            {
                i
                for token in (ref_token, actual_token)
                for i in range(max(0, token - 1), min(vocab, token + 2))
            }
        )
        _chain_emit(
            "cdf",
            b=b,
            values_exact=torch.equal(values[0], val),
            reference_token=ref_token,
            actual_token=actual_token,
            isolated_native_token=isolated_token,
            traced_native_token=traced_token,
            trace_matches_actual=traced_token == actual_token == isolated_token,
            coin=float(coin.item()),
            reference_sum=float(ref_sum.item()),
            native_sum=float(native_sum[0].item()),
            reference_threshold=float(rt.item()),
            native_threshold=float(ntarget.item()),
            cdf_max_abs_diff=float((ref_cdf - trace).abs().max().item()),
            mixed_tokens=mixed,
            near_tokens=[
                dict(
                    index=i,
                    torch_cdf=float(ref_cdf[i].item()),
                    native_cdf=float(trace[i].item()),
                    value=float(val[i].item()),
                )
                for i in around
            ],
        )
