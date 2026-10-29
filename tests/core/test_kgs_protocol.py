"""Per-case invocation evidence, including failures and bypasses."""
from types import SimpleNamespace

from tests import conftest
from flaggems_sglang.dynamic_registry import DynamicOpOverride


def test_call_count_snapshot_cannot_mutate_registry():
    registry = DynamicOpOverride()
    registry._call_counts['flaggems_sglang.op'] = 2
    snapshot = registry.call_counts()
    snapshot['flaggems_sglang.op'] = 99
    assert registry.call_counts()['flaggems_sglang.op'] == 2


def test_each_correctness_case_records_only_its_own_calls():
    registry = DynamicOpOverride()
    registry._call_counts['flaggems_sglang.op'] = 5
    config = SimpleNamespace(_override_registry=registry)
    for name, increments in [('called', 2), ('bypassed', 0)]:
        conftest.TEST_RESULTS[name] = {}
        hook = conftest.pytest_runtest_call(SimpleNamespace(config=config, nodeid=name))
        next(hook)
        registry._call_counts['flaggems_sglang.op'] += increments
        try:
            next(hook)
        except StopIteration:
            pass
        assert conftest.TEST_RESULTS[name]['candidate_calls'] == {'flaggems_sglang.op': increments}
