import copy
import pytest
from scripts.complete_priority_test import validate_report


def test_requested_seeds_and_reuse():
    from scripts.complete_priority_test import TEST_SEEDS
    assert TEST_SEEDS == (1, 7, 42)
    assert [s for s in TEST_SEEDS if s != 42] == [1, 7]


def test_coverage_configuration_and_provenance():
    configs = [dict(name='pure')]
    report = dict(split='test', indices=list(range(2000)), variants=dict(pure=dict(config=configs[0])),
                  checkpoints={}, data_sha256={}, sampling={}, batch_size=16)
    validate_report(report, configs, report)
    bad = copy.deepcopy(report)
    bad['indices'][-1] = 0
    with pytest.raises(ValueError):
        validate_report(bad, configs)
    bad = copy.deepcopy(report)
    bad['checkpoints'] = {'changed': 'hash'}
    with pytest.raises(ValueError):
        validate_report(bad, configs, report)