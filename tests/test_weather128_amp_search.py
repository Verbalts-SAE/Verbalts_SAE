import numpy as np

from scripts.run_weather128_amp_search import background_partition, next_amplitude


def test_target_includes_boundaries():
    for accr in (.4, .5, .6):
        assert next_amplitude([dict(amplitude=.5, accr=accr)]) is None


def test_amplitude_bounded_and_bracketed():
    # too easy -> step down
    assert next_amplitude([dict(amplitude=.5, accr=.9)]) == .4
    # too hard -> step up
    assert next_amplitude([dict(amplitude=.5, accr=.3)]) == .6
    # floor respected
    assert next_amplitude([dict(amplitude=.4, accr=.9)]) is None
    # ceiling respected
    assert next_amplitude([dict(amplitude=.7, accr=.3)]) is None
    # bracketing between easy/hard
    assert next_amplitude([dict(amplitude=.5, accr=.9), dict(amplitude=.4, accr=.3)]) == .45
    # inverted bracket cannot be resolved
    assert next_amplitude([dict(amplitude=.4, accr=.9), dict(amplitude=.5, accr=.3)]) is None


def test_partition_keeps_backgrounds_disjoint_and_all_combinations():
    ids = np.repeat(np.arange(50), 81)
    search, confirm = background_partition(ids)
    assert len(search) == 30 * 81 and len(confirm) == 20 * 81
    assert not set(ids[search]) & set(ids[confirm])
    np.testing.assert_array_equal(np.sort(np.r_[search, confirm]), np.arange(len(ids)))
    np.testing.assert_array_equal(search, background_partition(ids)[0])
