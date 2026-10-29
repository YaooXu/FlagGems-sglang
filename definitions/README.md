# Operator definitions

Store one structured definition per SGLang operator in
`definitions/<operator>.json`, following the directory layout of
[FlagGems kernelgen-dev](https://github.com/flagos-ai/FlagGems/tree/kernelgen-dev/definitions).

Each operator suite includes three corresponding files:

- `tests/test_<operator>.py`: correctness tests.
- `benchmark/test_<operator>.py`: performance benchmarks.
- `definitions/<operator>.json`: operator interface and semantic metadata.

Use the exact operator name in the JSON filename, including leading
underscores. Pytest marker naming is a separate convention.

Add the definition alongside its tests and benchmark in the same PR, targeting
`kernelgen-dev`. Keep it aligned with the tested parameters, outputs, mutations
and aliases whenever the operator contract changes.
