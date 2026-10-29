# Runnable Qwen3.6 mRoPE example

This example needs the repository's usual Torch/Triton/pytest environment.
It does **not** require installing SGLang, sglang-kernel, or compiling their
C++/CUDA extensions. Run the following commands from the repository root:

```sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
CANDIDATE=examples/candidates/triton_mrope_fused_sglang.py
OVERRIDE="triton_mrope_fused:$CANDIDATE:triton_mrope_fused"

# Real native Triton candidate versus the portable Torch oracle, all five cases.
pytest -q tests/test_triton_mrope_fused.py -m triton_mrope_fused --override "$OVERRIDE"
pytest -q tests/test_triton_mrope_fused.py -m triton_mrope_fused --override "$OVERRIDE" --ref cpu

# Enumerate cases without allocating tensors or invoking a candidate/timer.
pytest -q benchmark/test_triton_mrope_fused.py -m triton_mrope_fused \
  --level core --list-cases --output mrope-cases.json

# Time the actual injected candidate on the four documented configurations.
pytest -q benchmark/test_triton_mrope_fused.py -m triton_mrope_fused \
  --override "$OVERRIDE" --level core --warmup 1 --iter 1 \
  --record json --output mrope-timing.json

# Check each correctness case and each comprehensive benchmark case reaches injection.
pytest -q tests/test_triton_mrope_fused.py -m triton_mrope_fused \
  --dummy-candidate triton_mrope_fused --dummy-report mrope-dummy-tests.json
pytest -q benchmark/test_triton_mrope_fused.py -m triton_mrope_fused \
  --level comprehensive --dummy-candidate triton_mrope_fused \
  --dummy-report mrope-dummy-bench.json

# Independently check that the tests can select their explicit reference.
pytest -q tests/test_triton_mrope_fused.py -m triton_mrope_fused --ref-as-candidate
pytest -q benchmark/test_triton_mrope_fused.py -m triton_mrope_fused \
  --level core --ref-as-candidate --warmup 1 --iter 1
```

To debug a downstream implementation, change only `CANDIDATE` and the final
function name in `OVERRIDE`. Its signature must match the Definition. In
ordinary mode a missing public candidate skips; there is no silent fallback
to the reference. Use an override or explicit reference mode to execute it.

For exact replay, copy a `case_id` from `mrope-cases.json` and pass it as
`--case-id '...'` to the benchmark command. Core lists four cases;
comprehensive lists five. Neither mode changes the existing test runners.

## Contract and cases

The [27B workload](https://infrawaves.feishu.cn/wiki/KNeSwTT8miLWWZkWlWzcc3i0nMc)
and [35B-A3B workload](https://infrawaves.feishu.cn/wiki/VInzwGiF1ipx5ckwgKjcq1hqnLe)
describe SGLang v0.5.11, TP=4:

| Case | BF16 Q | BF16 K | INT64 positions |
| --- | --- | --- | --- |
| 27B prefill | `[20,1536]` | `[20,256]` | `[3,20]` |
| 27B decode | `[1,1536]` | `[1,256]` | `[3,1]`, value20 |
| 35B prefill | `[20,1024]` | `[20,256]` | `[3,20]` |
| 35B decode | `[1,1024]` | `[1,256]` | `[3,1]`, value20 |

Common: BF16 cos/sin cache `[262400,64]`, head_size256, rotary_dim64,
mrope_section `[11,11,10]`, interleaved=True, GLM=False, Neox=True,
axis_map=None. The function returns **None** and modifies Q/K in place.
Their last192 channels and all other inputs must stay exactly unchanged.

The fifth correctness/comprehensive case uses three tokens with distinct
THW positions and position stride `[512,1]`. This is an additional boundary
control, not a claimed model trace. The documents do not provide actual
tensors or numerical thresholds: synthetic inputs use an isolated seed0,
BF16 activations from FP32 normal(std0.5), and BF16 cos/sin of FP32
normal(std1) angles. Prefill positions use `arange(T)`; the extra case adds
THW offsets `[0,7,14]`. These recipes are test construction choices.

The portable reference is Torch composition. Q/K comparisons use engineering
thresholds atol=0.002, rtol=0.016; dtype/device/layout/storage, return None,
read-only inputs and unrotated channels are checked exactly. Reference and
candidate receive independent, stride-preserving input buffers. Explicit
reference selection uses PR #244's ordinary `get_candidate` interface.

## Native provenance and timing scope

The example candidate contains the original `_triton_mrope_forward_fused`
kernel and `triton_mrope_fused` wrapper from
[SGLang v0.5.11](https://github.com/sgl-project/sglang/blob/v0.5.11/python/sglang/srt/layers/rotary_embedding/triton_kernels.py),
Git blob `0a8dc2c33c7bbb2acb7e02b10796d210c5c079fc`.
Only formatting and attribution were added; compute function ASTs match the
source. No SGLang package bootstrap or compiled extension is imported.
The existing `mrope_fused` operator has a different ABI and remains unchanged.

Native candidate versus Torch is an independent numerical comparison;
reference-as-candidate validates only the harness. Benchmark JSON produced
above is a local diagnostic timing report. Formal cross-machine performance
uses separately collected NVIDIA artifacts through the existing `--base-data`
workflow; no formal H800 baseline is supplied by this example.
