import numpy as np

from scripts.run_weather128_difficulty import background_partition, next_ratio


def test_target_includes_boundaries():
    for accr in (.4, .5, .6):
        assert next_ratio([dict(ratio=.45, accr=accr)]) is None


def test_difficulty_bounded_and_bracketed():
    assert next_ratio([dict(ratio=.45, accr=.9)]) == .6
    assert next_ratio([dict(ratio=1.2, accr=.9)]) == 1.2
    assert next_ratio([dict(ratio=.45, accr=.9), dict(ratio=.6, accr=.3)]) == .525
    assert next_ratio([dict(ratio=.6, accr=.9), dict(ratio=.45, accr=.3)]) is None


def test_partition_keeps_backgrounds_disjoint_and_all_combinations():
    ids = np.repeat(np.arange(50), 81)
    search, confirm = background_partition(ids)
    assert len(search) == 30 * 81 and len(confirm) == 20 * 81
    assert not set(ids[search]) & set(ids[confirm])
    np.testing.assert_array_equal(np.sort(np.r_[search, confirm]), np.arange(len(ids)))
    np.testing.assert_array_equal(search, background_partition(ids)[0])