import numpy as np

from scripts.run_weather128_fine_search import background_partition, next_amplitude, AMPLITUDES


def test_target_includes_boundaries():
    for accr in (.4, .5, .6):
        assert next_amplitude([dict(amplitude=.8, accr=accr)]) is None


def test_amplitude_bounded_and_bracketed():
    # too easy -> step down along the fixed ladder
    assert next_amplitude([dict(amplitude=1.2, accr=.9)]) == .8
    assert next_amplitude([dict(amplitude=.8, accr=.9)]) == .5
    assert next_amplitude([dict(amplitude=.5, accr=.9)]) == .35
    # floor respected
    assert next_amplitude([dict(amplitude=.35, accr=.9)]) is None
    # too hard -> step up along the ladder
    assert next_amplitude([dict(amplitude=.5, accr=.3)]) == .8
    # ceiling respected
    assert next_amplitude([dict(amplitude=1.2, accr=.3)]) is None
    # bracketing between easy/hard takes the midpoint before the ladder
    assert next_amplitude([dict(amplitude=1.2, accr=.9), dict(amplitude=.5, accr=.3)]) == .85
    # inverted bracket cannot be resolved
    assert next_amplitude([dict(amplitude=.5, accr=.9), dict(amplitude=1.2, accr=.3)]) is None
    # never revisits a tested amplitude
    assert next_amplitude([dict(amplitude=1.2, accr=.9), dict(amplitude=.8, accr=.9)]) == .5


def test_partition_keeps_backgrounds_disjoint_and_bounded():
    ids = np.repeat(np.arange(50), 625)
    search, confirm = background_partition(ids, 3000)
    assert len(search) == 3000 and len(confirm) == 3000
    assert not set(ids[search]) & set(ids[confirm])
    assert len(np.unique(search)) == len(search) and len(np.unique(confirm)) == len(confirm)
    np.testing.assert_array_equal(search, background_partition(ids, 3000)[0])


def test_partition_respects_sample_size_floor():
    ids = np.repeat(np.arange(50), 625)
    search, confirm = background_partition(ids, 999)
    assert len(search) == 999 and len(confirm) == 999
    assert not set(ids[search]) & set(ids[confirm])
